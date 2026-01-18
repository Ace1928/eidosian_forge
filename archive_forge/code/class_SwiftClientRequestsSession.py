import requests
from requests.sessions import merge_setting, merge_hooks
from requests.structures import CaseInsensitiveDict
class SwiftClientRequestsSession(requests.Session):

    def prepare_request(self, request):
        p = SwiftClientPreparedRequest()
        headers = merge_setting(request.headers, self.headers, dict_class=CaseInsensitiveDict)
        p.prepare(method=request.method.upper(), url=request.url, files=request.files, data=request.data, json=request.json, headers=headers, params=merge_setting(request.params, self.params), auth=merge_setting(request.auth, self.auth), cookies=None, hooks=merge_hooks(request.hooks, self.hooks))
        return p