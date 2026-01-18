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
class TweetWriter(TweetHandlerI):
    """
    Handle data by writing it to a file.
    """

    def __init__(self, limit=2000, upper_date_limit=None, lower_date_limit=None, fprefix='tweets', subdir='twitter-files', repeat=False, gzip_compress=False):
        """
        The difference between the upper and lower date limits depends on
        whether Tweets are coming in an ascending date order (i.e. when
        streaming) or descending date order (i.e. when searching past Tweets).

        :param int limit: number of data items to process in the current        round of processing.

        :param tuple upper_date_limit: The date at which to stop collecting new        data. This should be entered as a tuple which can serve as the        argument to `datetime.datetime`. E.g. `upper_date_limit=(2015, 4, 1, 12,        40)` for 12:30 pm on April 1 2015.

        :param tuple lower_date_limit: The date at which to stop collecting new        data. See `upper_data_limit` for formatting.

        :param str fprefix: The prefix to use in creating file names for Tweet        collections.

        :param str subdir: The name of the directory where Tweet collection        files should be stored.

        :param bool repeat: flag to determine whether multiple files should be        written. If `True`, the length of each file will be set by the value        of `limit`. See also :py:func:`handle`.

        :param gzip_compress: if `True`, output files are compressed with gzip.
        """
        self.fprefix = fprefix
        self.subdir = guess_path(subdir)
        self.gzip_compress = gzip_compress
        self.fname = self.timestamped_file()
        self.repeat = repeat
        self.output = None
        TweetHandlerI.__init__(self, limit, upper_date_limit, lower_date_limit)

    def timestamped_file(self):
        """
        :return: timestamped file name
        :rtype: str
        """
        subdir = self.subdir
        fprefix = self.fprefix
        if subdir:
            if not os.path.exists(subdir):
                os.mkdir(subdir)
        fname = os.path.join(subdir, fprefix)
        fmt = '%Y%m%d-%H%M%S'
        timestamp = datetime.datetime.now().strftime(fmt)
        if self.gzip_compress:
            suffix = '.gz'
        else:
            suffix = ''
        outfile = f'{fname}.{timestamp}.json{suffix}'
        return outfile

    def handle(self, data):
        """
        Write Twitter data as line-delimited JSON into one or more files.

        :return: return `False` if processing should cease, otherwise return `True`.
        :param data: tweet object returned by Twitter API
        """
        if self.startingup:
            if self.gzip_compress:
                self.output = gzip.open(self.fname, 'w')
            else:
                self.output = open(self.fname, 'w')
            print(f'Writing to {self.fname}')
        json_data = json.dumps(data)
        if self.gzip_compress:
            self.output.write((json_data + '\n').encode('utf-8'))
        else:
            self.output.write(json_data + '\n')
        self.check_date_limit(data)
        if self.do_stop:
            return
        self.startingup = False

    def on_finish(self):
        print(f'Written {self.counter} Tweets')
        if self.output:
            self.output.close()

    def do_continue(self):
        if self.repeat == False:
            return TweetHandlerI.do_continue(self)
        if self.do_stop:
            return False
        if self.counter == self.limit:
            self._restart_file()
        return True

    def _restart_file(self):
        self.on_finish()
        self.fname = self.timestamped_file()
        self.startingup = True
        self.counter = 0