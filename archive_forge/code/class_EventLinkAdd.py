import logging
from os_ken.controller import handler
from os_ken.controller import event
class EventLinkAdd(EventLinkBase):

    def __init__(self, link):
        super(EventLinkAdd, self).__init__(link)