import random
import email.message
import pyzor
class PingRequest(ClientSideRequest):
    op = 'ping'