from twisted.web import resource
def _rewrite(self, request):
    for rewriteRule in self.rewriteRules:
        rewriteRule(request)