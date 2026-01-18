from urllib.parse import urljoin
from twisted.web import resource, server, static, util
class NoMetaRefreshRedirect(util.Redirect):

    def render(self, request):
        content = util.Redirect.render(self, request)
        return content.replace(b'http-equiv="refresh"', b'http-no-equiv="do-not-refresh-me"')