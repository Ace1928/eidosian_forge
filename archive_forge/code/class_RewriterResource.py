from twisted.web import resource
class RewriterResource(resource.Resource):

    def __init__(self, orig, *rewriteRules):
        resource.Resource.__init__(self)
        self.resource = orig
        self.rewriteRules = list(rewriteRules)

    def _rewrite(self, request):
        for rewriteRule in self.rewriteRules:
            rewriteRule(request)

    def getChild(self, path, request):
        request.postpath.insert(0, path)
        request.prepath.pop()
        self._rewrite(request)
        path = request.postpath.pop(0)
        request.prepath.append(path)
        return self.resource.getChildWithDefault(path, request)

    def render(self, request):
        self._rewrite(request)
        return self.resource.render(request)