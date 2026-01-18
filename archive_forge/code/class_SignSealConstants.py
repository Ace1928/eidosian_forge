class SignSealConstants(object):
    CLIENT_SIGNING = b'session key to client-to-server signing key magic constant\x00'
    SERVER_SIGNING = b'session key to server-to-client signing key magic constant\x00'
    CLIENT_SEALING = b'session key to client-to-server sealing key magic constant\x00'
    SERVER_SEALING = b'session key to server-to-client sealing key magic constant\x00'