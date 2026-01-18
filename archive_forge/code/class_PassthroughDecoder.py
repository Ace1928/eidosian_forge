import csv
import json
import logging
class PassthroughDecoder(object):

    def decode(self, x):
        return x