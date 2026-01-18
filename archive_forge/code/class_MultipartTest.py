import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy.test import helper
class MultipartTest(helper.CPWebCase):
    setup_server = staticmethod(setup_server)

    def test_multipart(self):
        text_part = ntou('This is the text version')
        html_part = ntou('<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">\n<html>\n<head>\n <meta content="text/html;charset=ISO-8859-1" http-equiv="Content-Type">\n</head>\n<body bgcolor="#ffffff" text="#000000">\n\nThis is the <strong>HTML</strong> version\n</body>\n</html>\n')
        body = '\r\n'.join(['--123456789', "Content-Type: text/plain; charset='ISO-8859-1'", 'Content-Transfer-Encoding: 7bit', '', text_part, '--123456789', "Content-Type: text/html; charset='ISO-8859-1'", '', html_part, '--123456789--'])
        headers = [('Content-Type', 'multipart/mixed; boundary=123456789'), ('Content-Length', str(len(body)))]
        self.getPage('/multipart', headers, 'POST', body)
        self.assertBody(repr([text_part, html_part]))

    def test_multipart_form_data(self):
        body = '\r\n'.join(['--X', 'Content-Disposition: form-data; name="foo"', '', 'bar', '--X', 'Content-Disposition: form-data; name="baz"', '', '111', '--X', 'Content-Disposition: form-data; name="baz"', '', '333', '--X--'])
        (self.getPage('/multipart_form_data', method='POST', headers=[('Content-Type', 'multipart/form-data;boundary=X'), ('Content-Length', str(len(body)))], body=body),)
        self.assertBody(repr([('baz', [ntou('111'), ntou('333')]), ('foo', ntou('bar'))]))