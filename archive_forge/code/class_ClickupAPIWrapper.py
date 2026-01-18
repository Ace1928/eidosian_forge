import json
import warnings
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, Union
import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
class ClickupAPIWrapper(BaseModel):
    """Wrapper for Clickup API."""
    access_token: Optional[str] = None
    team_id: Optional[str] = None
    space_id: Optional[str] = None
    folder_id: Optional[str] = None
    list_id: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    @classmethod
    def get_access_code_url(cls, oauth_client_id: str, redirect_uri: str='https://google.com') -> str:
        """Get the URL to get an access code."""
        url = f'https://app.clickup.com/api?client_id={oauth_client_id}'
        return f'{url}&redirect_uri={redirect_uri}'

    @classmethod
    def get_access_token(cls, oauth_client_id: str, oauth_client_secret: str, code: str) -> Optional[str]:
        """Get the access token."""
        url = f'{DEFAULT_URL}/oauth/token'
        params = {'client_id': oauth_client_id, 'client_secret': oauth_client_secret, 'code': code}
        response = requests.post(url, params=params)
        data = response.json()
        if 'access_token' not in data:
            print(f'Error: {data}')
            if 'ECODE' in data and data['ECODE'] == 'OAUTH_014':
                url = ClickupAPIWrapper.get_access_code_url(oauth_client_id)
                print('You already used this code once. Generate a new one.', f'Our best guess for the url to get a new code is:\n{url}')
            return None
        return data['access_token']

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values['access_token'] = get_from_dict_or_env(values, 'access_token', 'CLICKUP_ACCESS_TOKEN')
        values['team_id'] = fetch_team_id(values['access_token'])
        values['space_id'] = fetch_space_id(values['team_id'], values['access_token'])
        values['folder_id'] = fetch_folder_id(values['space_id'], values['access_token'])
        values['list_id'] = fetch_list_id(values['space_id'], values['folder_id'], values['access_token'])
        return values

    def attempt_parse_teams(self, input_dict: dict) -> Dict[str, List[dict]]:
        """Parse appropriate content from the list of teams."""
        parsed_teams: Dict[str, List[dict]] = {'teams': []}
        for team in input_dict['teams']:
            try:
                team = parse_dict_through_component(team, Team, fault_tolerant=False)
                parsed_teams['teams'].append(team)
            except Exception as e:
                warnings.warn(f'Error parsing a team {e}')
        return parsed_teams

    def get_headers(self) -> Mapping[str, Union[str, bytes]]:
        """Get the headers for the request."""
        if not isinstance(self.access_token, str):
            raise TypeError(f'Access Token: {self.access_token}, must be str.')
        headers = {'Authorization': str(self.access_token), 'Content-Type': 'application/json'}
        return headers

    def get_default_params(self) -> Dict:
        return {'archived': 'false'}

    def get_authorized_teams(self) -> Dict[Any, Any]:
        """Get all teams for the user."""
        url = f'{DEFAULT_URL}/team'
        response = requests.get(url, headers=self.get_headers())
        data = response.json()
        parsed_teams = self.attempt_parse_teams(data)
        return parsed_teams

    def get_folders(self) -> Dict:
        """
        Get all the folders for the team.
        """
        url = f'{DEFAULT_URL}/team/' + str(self.team_id) + '/space'
        params = self.get_default_params()
        response = requests.get(url, headers=self.get_headers(), params=params)
        return {'response': response}

    def get_task(self, query: str, fault_tolerant: bool=True) -> Dict:
        """
        Retrieve a specific task.
        """
        params, error = load_query(query, fault_tolerant=True)
        if params is None:
            return {'Error': error}
        url = f'{DEFAULT_URL}/task/{params['task_id']}'
        params = {'custom_task_ids': 'true', 'team_id': self.team_id, 'include_subtasks': 'true'}
        response = requests.get(url, headers=self.get_headers(), params=params)
        data = response.json()
        parsed_task = parse_dict_through_component(data, Task, fault_tolerant=fault_tolerant)
        return parsed_task

    def get_lists(self) -> Dict:
        """
        Get all available lists.
        """
        url = f'{DEFAULT_URL}/folder/{self.folder_id}/list'
        params = self.get_default_params()
        response = requests.get(url, headers=self.get_headers(), params=params)
        return {'response': response}

    def query_tasks(self, query: str) -> Dict:
        """
        Query tasks that match certain fields
        """
        params, error = load_query(query, fault_tolerant=True)
        if params is None:
            return {'Error': error}
        url = f'{DEFAULT_URL}/list/{params['list_id']}/task'
        params = self.get_default_params()
        response = requests.get(url, headers=self.get_headers(), params=params)
        return {'response': response}

    def get_spaces(self) -> Dict:
        """
        Get all spaces for the team.
        """
        url = f'{DEFAULT_URL}/team/{self.team_id}/space'
        response = requests.get(url, headers=self.get_headers(), params=self.get_default_params())
        data = response.json()
        parsed_spaces = parse_dict_through_component(data, Space, fault_tolerant=True)
        return parsed_spaces

    def get_task_attribute(self, query: str) -> Dict:
        """
        Update an attribute of a specified task.
        """
        task = self.get_task(query, fault_tolerant=True)
        params, error = load_query(query, fault_tolerant=True)
        if not isinstance(params, dict):
            return {'Error': error}
        if params['attribute_name'] not in task:
            return {'Error': f'attribute_name = {params['attribute_name']} was not \nfound in task keys {task.keys()}. Please call again with one of the key names.'}
        return {params['attribute_name']: task[params['attribute_name']]}

    def update_task(self, query: str) -> Dict:
        """
        Update an attribute of a specified task.
        """
        query_dict, error = load_query(query, fault_tolerant=True)
        if query_dict is None:
            return {'Error': error}
        url = f'{DEFAULT_URL}/task/{query_dict['task_id']}'
        params = {'custom_task_ids': 'true', 'team_id': self.team_id, 'include_subtasks': 'true'}
        headers = self.get_headers()
        payload = {query_dict['attribute_name']: query_dict['value']}
        response = requests.put(url, headers=headers, params=params, json=payload)
        return {'response': response}

    def update_task_assignees(self, query: str) -> Dict:
        """
        Add or remove assignees of a specified task.
        """
        query_dict, error = load_query(query, fault_tolerant=True)
        if query_dict is None:
            return {'Error': error}
        for user in query_dict['users']:
            if not isinstance(user, int):
                return {'Error': f'All users must be integers, not strings!\n"Got user {user} if type {type(user)}'}
        url = f'{DEFAULT_URL}/task/{query_dict['task_id']}'
        headers = self.get_headers()
        if query_dict['operation'] == 'add':
            assigne_payload = {'add': query_dict['users'], 'rem': []}
        elif query_dict['operation'] == 'rem':
            assigne_payload = {'add': [], 'rem': query_dict['users']}
        else:
            raise ValueError(f'Invalid operation ({query_dict['operation']}). ', "Valid options ['add', 'rem'].")
        params = {'custom_task_ids': 'true', 'team_id': self.team_id, 'include_subtasks': 'true'}
        payload = {'assignees': assigne_payload}
        response = requests.put(url, headers=headers, params=params, json=payload)
        return {'response': response}

    def create_task(self, query: str) -> Dict:
        """
        Creates a new task.
        """
        query_dict, error = load_query(query, fault_tolerant=True)
        if query_dict is None:
            return {'Error': error}
        list_id = self.list_id
        url = f'{DEFAULT_URL}/list/{list_id}/task'
        params = {'custom_task_ids': 'true', 'team_id': self.team_id}
        payload = extract_dict_elements_from_component_fields(query_dict, Task)
        headers = self.get_headers()
        response = requests.post(url, json=payload, headers=headers, params=params)
        data: Dict = response.json()
        return parse_dict_through_component(data, Task, fault_tolerant=True)

    def create_list(self, query: str) -> Dict:
        """
        Creates a new list.
        """
        query_dict, error = load_query(query, fault_tolerant=True)
        if query_dict is None:
            return {'Error': error}
        location = self.folder_id if self.folder_id else self.space_id
        url = f'{DEFAULT_URL}/folder/{location}/list'
        payload = extract_dict_elements_from_component_fields(query_dict, Task)
        headers = self.get_headers()
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()
        parsed_list = parse_dict_through_component(data, CUList, fault_tolerant=True)
        if 'id' in parsed_list:
            self.list_id = parsed_list['id']
        return parsed_list

    def create_folder(self, query: str) -> Dict:
        """
        Creates a new folder.
        """
        query_dict, error = load_query(query, fault_tolerant=True)
        if query_dict is None:
            return {'Error': error}
        space_id = self.space_id
        url = f'{DEFAULT_URL}/space/{space_id}/folder'
        payload = {'name': query_dict['name']}
        headers = self.get_headers()
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()
        if 'id' in data:
            self.list_id = data['id']
        return data

    def run(self, mode: str, query: str) -> str:
        """Run the API."""
        if mode == 'get_task':
            output = self.get_task(query)
        elif mode == 'get_task_attribute':
            output = self.get_task_attribute(query)
        elif mode == 'get_teams':
            output = self.get_authorized_teams()
        elif mode == 'create_task':
            output = self.create_task(query)
        elif mode == 'create_list':
            output = self.create_list(query)
        elif mode == 'create_folder':
            output = self.create_folder(query)
        elif mode == 'get_lists':
            output = self.get_lists()
        elif mode == 'get_folders':
            output = self.get_folders()
        elif mode == 'get_spaces':
            output = self.get_spaces()
        elif mode == 'update_task':
            output = self.update_task(query)
        elif mode == 'update_task_assignees':
            output = self.update_task_assignees(query)
        else:
            output = {'ModeError': f'Got unexpected mode {mode}.'}
        try:
            return json.dumps(output)
        except Exception:
            return str(output)