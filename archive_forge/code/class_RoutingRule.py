class RoutingRule(object):
    """Represents a single routing rule.

    There are convenience methods to making creating rules
    more concise::

        rule = RoutingRule.when(key_prefix='foo/').then_redirect('example.com')

    :ivar condition: Describes condition that must be met for the
        specified redirect to apply.

    :ivar redirect: Specifies redirect behavior.  You can redirect requests to
        another host, to another page, or with another protocol. In the event
        of an error, you can can specify a different error code to return.

    """

    def __init__(self, condition=None, redirect=None):
        self.condition = condition
        self.redirect = redirect

    def startElement(self, name, attrs, connection):
        if name == 'Condition':
            return self.condition
        elif name == 'Redirect':
            return self.redirect

    def endElement(self, name, value, connection):
        pass

    def to_xml(self):
        parts = []
        if self.condition:
            parts.append(self.condition.to_xml())
        if self.redirect:
            parts.append(self.redirect.to_xml())
        return tag('RoutingRule', '\n'.join(parts))

    @classmethod
    def when(cls, key_prefix=None, http_error_code=None):
        return cls(Condition(key_prefix=key_prefix, http_error_code=http_error_code), None)

    def then_redirect(self, hostname=None, protocol=None, replace_key=None, replace_key_prefix=None, http_redirect_code=None):
        self.redirect = Redirect(hostname=hostname, protocol=protocol, replace_key=replace_key, replace_key_prefix=replace_key_prefix, http_redirect_code=http_redirect_code)
        return self