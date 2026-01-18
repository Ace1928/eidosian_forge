import base64
import socket
import struct
import sys
def __getauthheader(self):
    auth = self.__proxy[4] + ':' + self.__proxy[5]
    return 'Proxy-Authorization: Basic ' + base64.b64encode(auth)