import time
import email
import socket
import logging
import functools
import collections
import pyzor.digest
import pyzor.account
import pyzor.message
import pyzor.hacks.py26
class CheckClientRunner(ClientRunner):

    def __init__(self, routine, r_count=0, wl_count=0):
        ClientRunner.__init__(self, routine)
        self.found_hit = False
        self.whitelisted = False
        self.hit_count = 0
        self.whitelist_count = 0
        self.r_count_found = r_count
        self.wl_count_clears = wl_count

    def handle_response(self, response, message):
        message += '%s\t' % str(response.head_tuple())
        if response.is_ok():
            self.hit_count = int(response['Count'])
            self.whitelist_count = int(response['WL-Count'])
            if self.whitelist_count > self.wl_count_clears:
                self.whitelisted = True
            elif self.hit_count > self.r_count_found:
                self.found_hit = True
            message += '%d\t%d' % (self.hit_count, self.whitelist_count)
        else:
            self.all_ok = False
        self.results.append(message + '\n')