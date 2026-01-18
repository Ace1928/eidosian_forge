import logging
from os_ken.controller import handler
from os_ken.controller import event
class EventLinkDelete(EventLinkBase):

    def __init__(self, link):
        super(EventLinkDelete, self).__init__(link)