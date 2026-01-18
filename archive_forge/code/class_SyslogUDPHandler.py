import os
import logging
import socketserver
class SyslogUDPHandler(socketserver.BaseRequestHandler):
    """ A handler """

    def handle(self):
        """ Handle data """
        data = bytes.decode(self.request[0].strip())
        print(f'{self.client_address[0]}: {str(data)}')
        logging.info(str(data))