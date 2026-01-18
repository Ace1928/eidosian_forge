from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import binary_type, PY3
from ansible.module_utils.six.moves.http_client import responses as http_responses
class ACMEProtocolException(ModuleFailException):

    def __init__(self, module, msg=None, info=None, response=None, content=None, content_json=None, extras=None):
        if content is None and content_json is None and (response is not None):
            try:
                if PY3 and response.closed:
                    raise TypeError
                content = response.read()
            except (AttributeError, TypeError):
                content = info.pop('body', None)
        if content_json is not None and (not isinstance(content_json, dict)):
            if content is None and isinstance(content_json, binary_type):
                content = content_json
            content_json = None
        if content_json is None and content is not None and (module is not None):
            try:
                content_json = module.from_json(to_text(content))
            except Exception as e:
                pass
        extras = extras or dict()
        if msg is None:
            msg = 'ACME request failed'
        add_msg = ''
        if info is not None:
            url = info['url']
            code = info['status']
            extras['http_url'] = url
            extras['http_status'] = code
            if code is not None and code >= 400 and (content_json is not None) and ('type' in content_json):
                if 'status' in content_json and content_json['status'] != code:
                    code_msg = 'status {problem_code} (HTTP status: {http_code})'.format(http_code=format_http_status(code), problem_code=content_json['status'])
                else:
                    code_msg = 'status {problem_code}'.format(problem_code=format_http_status(code))
                    if code == -1 and info.get('msg'):
                        code_msg = 'error: {msg}'.format(msg=info['msg'])
                subproblems = content_json.pop('subproblems', None)
                add_msg = ' {problem}.'.format(problem=format_error_problem(content_json))
                extras['problem'] = content_json
                extras['subproblems'] = subproblems or []
                if subproblems is not None:
                    add_msg = '{add_msg} Subproblems:'.format(add_msg=add_msg)
                    for index, problem in enumerate(subproblems):
                        add_msg = '{add_msg}\n({index}) {problem}.'.format(add_msg=add_msg, index=index, problem=format_error_problem(problem, subproblem_prefix='{0}.'.format(index)))
            else:
                code_msg = 'HTTP status {code}'.format(code=format_http_status(code))
                if code == -1 and info.get('msg'):
                    code_msg = 'error: {msg}'.format(msg=info['msg'])
                if content_json is not None:
                    add_msg = ' The JSON error result: {content}'.format(content=content_json)
                elif content is not None:
                    add_msg = ' The raw error result: {content}'.format(content=to_text(content))
            msg = '{msg} for {url} with {code}'.format(msg=msg, url=url, code=code_msg)
        elif content_json is not None:
            add_msg = ' The JSON result: {content}'.format(content=content_json)
        elif content is not None:
            add_msg = ' The raw result: {content}'.format(content=to_text(content))
        super(ACMEProtocolException, self).__init__('{msg}.{add_msg}'.format(msg=msg, add_msg=add_msg), **extras)
        self.problem = {}
        self.subproblems = []
        for k, v in extras.items():
            setattr(self, k, v)