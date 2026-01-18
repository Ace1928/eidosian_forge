import operator
import os.path
import sys
import unittest
import testscenarios
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_protocol
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import ofproto_v1_4
from os_ken.ofproto import ofproto_v1_5
from os_ken import exception
import json
@staticmethod
def _msg_to_jsondict(msg):
    return msg.to_jsondict()