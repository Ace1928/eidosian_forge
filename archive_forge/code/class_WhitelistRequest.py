import random
import email.message
import pyzor
class WhitelistRequest(SimpleDigestSpecBasedRequest):
    op = 'whitelist'