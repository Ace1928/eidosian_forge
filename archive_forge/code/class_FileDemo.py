from myFile.file and write() somewhere else.
import os
import os.path
import cherrypy
from cherrypy.lib import static
class FileDemo(object):

    @cherrypy.expose
    def index(self):
        return '\n        <html><body>\n            <h2>Upload a file</h2>\n            <form action="upload" method="post" enctype="multipart/form-data">\n            filename: <input type="file" name="myFile" /><br />\n            <input type="submit" />\n            </form>\n            <h2>Download a file</h2>\n            <a href=\'download\'>This one</a>\n        </body></html>\n        '

    @cherrypy.expose
    def upload(self, myFile):
        out = '<html>\n        <body>\n            myFile length: %s<br />\n            myFile filename: %s<br />\n            myFile mime-type: %s\n        </body>\n        </html>'
        size = 0
        while True:
            data = myFile.file.read(8192)
            if not data:
                break
            size += len(data)
        return out % (size, myFile.filename, myFile.content_type)

    @cherrypy.expose
    def download(self):
        path = os.path.join(absDir, 'pdf_file.pdf')
        return static.serve_file(path, 'application/x-download', 'attachment', os.path.basename(path))