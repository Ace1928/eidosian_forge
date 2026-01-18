import sys
import os
import boto
import optparse
import copy
import boto.exception
import boto.roboto.awsqueryservice
import bdb
import traceback
class RequiredParamError(boto.exception.BotoClientError):

    def __init__(self, required):
        self.required = required
        s = 'Required parameters are missing: %s' % self.required
        super(RequiredParamError, self).__init__(s)