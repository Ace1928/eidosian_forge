from paste.request import construct_url
class PonyMiddleware(object):

    def __init__(self, application):
        self.application = application

    def __call__(self, environ, start_response):
        path_info = environ.get('PATH_INFO', '')
        if path_info == '/pony':
            url = construct_url(environ, with_query_string=False)
            if 'horn' in environ.get('QUERY_STRING', ''):
                data = UNICORN
                link = 'remove horn!'
            else:
                data = PONY
                url += '?horn'
                link = 'add horn!'
            msg = data.decode('base64').decode('zlib')
            msg = '<pre>%s\n<a href="%s">%s</a></pre>' % (msg, url, link)
            start_response('200 OK', [('content-type', 'text/html')])
            return [msg]
        else:
            return self.application(environ, start_response)