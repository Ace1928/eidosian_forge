import cherrypy
from cherrypy.lib.reprconf import attributes
from cherrypy._cpcompat import text_or_bytes
from cherrypy.process.servers import ServerAdapter
Return the base for this server.

        e.i. scheme://host[:port] or sock file
        