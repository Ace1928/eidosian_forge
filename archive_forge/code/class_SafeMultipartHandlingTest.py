import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy.test import helper
class SafeMultipartHandlingTest(helper.CPWebCase):
    setup_server = staticmethod(setup_server)

    def test_Flash_Upload(self):
        headers = [('Accept', 'text/*'), ('Content-Type', 'multipart/form-data; boundary=----------KM7Ij5cH2KM7Ef1gL6ae0ae0cH2gL6'), ('User-Agent', 'Shockwave Flash'), ('Host', 'www.example.com:54583'), ('Content-Length', '499'), ('Connection', 'Keep-Alive'), ('Cache-Control', 'no-cache')]
        filedata = b'<?xml version="1.0" encoding="UTF-8"?>\r\n<projectDescription>\r\n</projectDescription>\r\n'
        body = b'------------KM7Ij5cH2KM7Ef1gL6ae0ae0cH2gL6\r\nContent-Disposition: form-data; name="Filename"\r\n\r\n.project\r\n------------KM7Ij5cH2KM7Ef1gL6ae0ae0cH2gL6\r\nContent-Disposition: form-data; name="Filedata"; filename=".project"\r\nContent-Type: application/octet-stream\r\n\r\n' + filedata + b'\r\n------------KM7Ij5cH2KM7Ef1gL6ae0ae0cH2gL6\r\nContent-Disposition: form-data; name="Upload"\r\n\r\nSubmit Query\r\n------------KM7Ij5cH2KM7Ef1gL6ae0ae0cH2gL6--'
        self.getPage('/flashupload', headers, 'POST', body)
        self.assertBody('Upload: Submit Query, Filename: .project, Filedata: %r' % filedata)