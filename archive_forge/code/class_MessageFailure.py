from django.contrib.messages import constants
from django.contrib.messages.storage import default_storage
class MessageFailure(Exception):
    pass