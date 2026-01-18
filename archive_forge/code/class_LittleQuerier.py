from __future__ import print_function
import boto
import time
import uuid
from threading import Thread
class LittleQuerier(object):
    """
    An object that manages a thread that keeps pulling down small
    objects from S3 and checking the answers until told to stop.
    """

    def __init__(self, bucket, small_names):
        self.running = True
        self.bucket = bucket
        self.small_names = small_names
        self.thread = spawn(self.run)

    def stop(self):
        self.running = False
        self.thread.join()

    def run(self):
        count = 0
        while self.running:
            i = count % 4
            key = self.bucket.get_key(self.small_names[i])
            expected = str(i)
            rh = {'response-content-type': 'small/' + str(i)}
            actual = key.get_contents_as_string(response_headers=rh).decode('utf-8')
            if expected != actual:
                print('AHA:', repr(expected), repr(actual))
            assert expected == actual
            count += 1