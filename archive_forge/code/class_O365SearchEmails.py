from typing import Any, Dict, List, Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Extra, Field
from langchain_community.tools.office365.base import O365BaseTool
from langchain_community.tools.office365.utils import UTC_FORMAT, clean_body
class O365SearchEmails(O365BaseTool):
    """Search email messages in Office 365.

    Free, but setup is required.
    """
    name: str = 'messages_search'
    args_schema: Type[BaseModel] = SearchEmailsInput
    description: str = 'Use this tool to search for email messages. The input must be a valid Microsoft Graph v1.0 $search query. The output is a JSON list of the requested resource.'

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    def _run(self, query: str, folder: str='', max_results: int=10, truncate: bool=True, run_manager: Optional[CallbackManagerForToolRun]=None, truncate_limit: int=150) -> List[Dict[str, Any]]:
        mailbox = self.account.mailbox()
        if folder != '':
            mailbox = mailbox.get_folder(folder_name=folder)
        query = mailbox.q().search(query)
        messages = mailbox.get_messages(limit=max_results, query=query)
        output_messages = []
        for message in messages:
            output_message = {}
            output_message['from'] = message.sender
            if truncate:
                output_message['body'] = message.body_preview[:truncate_limit]
            else:
                output_message['body'] = clean_body(message.body)
            output_message['subject'] = message.subject
            output_message['date'] = message.modified.strftime(UTC_FORMAT)
            output_message['to'] = []
            for recipient in message.to._recipients:
                output_message['to'].append(str(recipient))
            output_message['cc'] = []
            for recipient in message.cc._recipients:
                output_message['cc'].append(str(recipient))
            output_message['bcc'] = []
            for recipient in message.bcc._recipients:
                output_message['bcc'].append(str(recipient))
            output_messages.append(output_message)
        return output_messages