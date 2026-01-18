class CORSRule(object):
    """
    CORS rule for a bucket.

    :ivar id: A unique identifier for the rule.  The ID value can be
        up to 255 characters long.  The IDs help you find a rule in
        the configuration.

    :ivar allowed_methods: An HTTP method that you want to allow the
        origin to execute.  Each CORSRule must identify at least one
        origin and one method. Valid values are:
        GET|PUT|HEAD|POST|DELETE

    :ivar allowed_origin: An origin that you want to allow cross-domain
        requests from. This can contain at most one * wild character.
        Each CORSRule must identify at least one origin and one method.
        The origin value can include at most one '*' wild character.
        For example, "http://*.example.com". You can also specify
        only * as the origin value allowing all origins cross-domain access.

    :ivar allowed_header: Specifies which headers are allowed in a
        pre-flight OPTIONS request via the
        Access-Control-Request-Headers header. Each header name
        specified in the Access-Control-Request-Headers header must
        have a corresponding entry in the rule. Amazon S3 will send
        only the allowed headers in a response that were requested.
        This can contain at most one * wild character.

    :ivar max_age_seconds: The time in seconds that your browser is to
        cache the preflight response for the specified resource.

    :ivar expose_header: One or more headers in the response that you
        want customers to be able to access from their applications
        (for example, from a JavaScript XMLHttpRequest object).  You
        add one ExposeHeader element in the rule for each header.
        """

    def __init__(self, allowed_method=None, allowed_origin=None, id=None, allowed_header=None, max_age_seconds=None, expose_header=None):
        if allowed_method is None:
            allowed_method = []
        self.allowed_method = allowed_method
        if allowed_origin is None:
            allowed_origin = []
        self.allowed_origin = allowed_origin
        self.id = id
        if allowed_header is None:
            allowed_header = []
        self.allowed_header = allowed_header
        self.max_age_seconds = max_age_seconds
        if expose_header is None:
            expose_header = []
        self.expose_header = expose_header

    def __repr__(self):
        return '<Rule: %s>' % self.id

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'ID':
            self.id = value
        elif name == 'AllowedMethod':
            self.allowed_method.append(value)
        elif name == 'AllowedOrigin':
            self.allowed_origin.append(value)
        elif name == 'AllowedHeader':
            self.allowed_header.append(value)
        elif name == 'MaxAgeSeconds':
            self.max_age_seconds = int(value)
        elif name == 'ExposeHeader':
            self.expose_header.append(value)
        else:
            setattr(self, name, value)

    def to_xml(self):
        s = '<CORSRule>'
        for allowed_method in self.allowed_method:
            s += '<AllowedMethod>%s</AllowedMethod>' % allowed_method
        for allowed_origin in self.allowed_origin:
            s += '<AllowedOrigin>%s</AllowedOrigin>' % allowed_origin
        for allowed_header in self.allowed_header:
            s += '<AllowedHeader>%s</AllowedHeader>' % allowed_header
        for expose_header in self.expose_header:
            s += '<ExposeHeader>%s</ExposeHeader>' % expose_header
        if self.max_age_seconds:
            s += '<MaxAgeSeconds>%d</MaxAgeSeconds>' % self.max_age_seconds
        if self.id:
            s += '<ID>%s</ID>' % self.id
        s += '</CORSRule>'
        return s