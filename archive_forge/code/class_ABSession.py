import getopt
import os
import re
import sys
import time
import cherrypy
from cherrypy import _cperror, _cpmodpy
from cherrypy.lib import httputil
class ABSession:
    """A session of 'ab', the Apache HTTP server benchmarking tool.

Example output from ab:

This is ApacheBench, Version 2.0.40-dev <$Revision: 1.121.2.1 $> apache-2.0
Copyright (c) 1996 Adam Twiss, Zeus Technology Ltd, http://www.zeustech.net/
Copyright (c) 1998-2002 The Apache Software Foundation, http://www.apache.org/

Benchmarking 127.0.0.1 (be patient)
Completed 100 requests
Completed 200 requests
Completed 300 requests
Completed 400 requests
Completed 500 requests
Completed 600 requests
Completed 700 requests
Completed 800 requests
Completed 900 requests


Server Software:        CherryPy/3.1beta
Server Hostname:        127.0.0.1
Server Port:            54583

Document Path:          /static/index.html
Document Length:        14 bytes

Concurrency Level:      10
Time taken for tests:   9.643867 seconds
Complete requests:      1000
Failed requests:        0
Write errors:           0
Total transferred:      189000 bytes
HTML transferred:       14000 bytes
Requests per second:    103.69 [#/sec] (mean)
Time per request:       96.439 [ms] (mean)
Time per request:       9.644 [ms] (mean, across all concurrent requests)
Transfer rate:          19.08 [Kbytes/sec] received

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   2.9      0      10
Processing:    20   94   7.3     90     130
Waiting:        0   43  28.1     40     100
Total:         20   95   7.3    100     130

Percentage of the requests served within a certain time (ms)
  50%    100
  66%    100
  75%    100
  80%    100
  90%    100
  95%    100
  98%    100
  99%    110
 100%    130 (longest request)
Finished 1000 requests
"""
    parse_patterns = [('complete_requests', 'Completed', b'^Complete requests:\\s*(\\d+)'), ('failed_requests', 'Failed', b'^Failed requests:\\s*(\\d+)'), ('requests_per_second', 'req/sec', b'^Requests per second:\\s*([0-9.]+)'), ('time_per_request_concurrent', 'msec/req', b'^Time per request:\\s*([0-9.]+).*concurrent requests\\)$'), ('transfer_rate', 'KB/sec', b'^Transfer rate:\\s*([0-9.]+)')]

    def __init__(self, path=SCRIPT_NAME + '/hello', requests=1000, concurrency=10):
        self.path = path
        self.requests = requests
        self.concurrency = concurrency

    def args(self):
        port = cherrypy.server.socket_port
        assert self.concurrency > 0
        assert self.requests > 0
        return '-k -n %s -c %s http://127.0.0.1:%s%s' % (self.requests, self.concurrency, port, self.path)

    def run(self):
        try:
            self.output = _cpmodpy.read_process(AB_PATH or 'ab', self.args())
        except Exception:
            print(_cperror.format_exc())
            raise
        for attr, name, pattern in self.parse_patterns:
            val = re.search(pattern, self.output, re.MULTILINE)
            if val:
                val = val.group(1)
                setattr(self, attr, val)
            else:
                setattr(self, attr, None)