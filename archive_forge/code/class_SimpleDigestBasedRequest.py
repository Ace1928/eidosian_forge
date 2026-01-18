import random
import email.message
import pyzor
class SimpleDigestBasedRequest(ClientSideRequest):

    def __init__(self, digest=None):
        ClientSideRequest.__init__(self)
        self.digest_count = 0
        if digest:
            self.add_digest(digest)

    def add_digest(self, digest):
        self.add_header('Op-Digest', digest)
        self.digest_count += 1