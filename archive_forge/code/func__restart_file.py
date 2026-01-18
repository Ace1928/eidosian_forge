import datetime
import gzip
import itertools
import json
import os
import time
import requests
from twython import Twython, TwythonStreamer
from twython.exceptions import TwythonError, TwythonRateLimitError
from nltk.twitter.api import BasicTweetHandler, TweetHandlerI
from nltk.twitter.util import credsfromfile, guess_path
def _restart_file(self):
    self.on_finish()
    self.fname = self.timestamped_file()
    self.startingup = True
    self.counter = 0