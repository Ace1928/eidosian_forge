import logging
import os
import sys
import threading
import time
import cherrypy
from cherrypy._json import json
class StatsTool(cherrypy.Tool):
    """Record various information about the current request."""

    def __init__(self):
        cherrypy.Tool.__init__(self, 'on_end_request', self.record_stop)

    def _setup(self):
        """Hook this tool into cherrypy.request.

        The standard CherryPy request object will automatically call this
        method when the tool is "turned on" in config.
        """
        if appstats.get('Enabled', False):
            cherrypy.Tool._setup(self)
            self.record_start()

    def record_start(self):
        """Record the beginning of a request."""
        request = cherrypy.serving.request
        if not hasattr(request.rfile, 'bytes_read'):
            request.rfile = ByteCountWrapper(request.rfile)
            request.body.fp = request.rfile
        r = request.remote
        appstats['Current Requests'] += 1
        appstats['Total Requests'] += 1
        appstats['Requests'][_get_threading_ident()] = {'Bytes Read': None, 'Bytes Written': None, 'Client': lambda s: '%s:%s' % (r.ip, r.port), 'End Time': None, 'Processing Time': proc_time, 'Request-Line': request.request_line, 'Response Status': None, 'Start Time': time.time()}

    def record_stop(self, uriset=None, slow_queries=1.0, slow_queries_count=100, debug=False, **kwargs):
        """Record the end of a request."""
        resp = cherrypy.serving.response
        w = appstats['Requests'][_get_threading_ident()]
        r = cherrypy.request.rfile.bytes_read
        w['Bytes Read'] = r
        appstats['Total Bytes Read'] += r
        if resp.stream:
            w['Bytes Written'] = 'chunked'
        else:
            cl = int(resp.headers.get('Content-Length', 0))
            w['Bytes Written'] = cl
            appstats['Total Bytes Written'] += cl
        w['Response Status'] = getattr(resp, 'output_status', resp.status).decode()
        w['End Time'] = time.time()
        p = w['End Time'] - w['Start Time']
        w['Processing Time'] = p
        appstats['Total Time'] += p
        appstats['Current Requests'] -= 1
        if debug:
            cherrypy.log('Stats recorded: %s' % repr(w), 'TOOLS.CPSTATS')
        if uriset:
            rs = appstats.setdefault('URI Set Tracking', {})
            r = rs.setdefault(uriset, {'Min': None, 'Max': None, 'Count': 0, 'Sum': 0, 'Avg': average_uriset_time})
            if r['Min'] is None or p < r['Min']:
                r['Min'] = p
            if r['Max'] is None or p > r['Max']:
                r['Max'] = p
            r['Count'] += 1
            r['Sum'] += p
        if slow_queries and p > slow_queries:
            sq = appstats.setdefault('Slow Queries', [])
            sq.append(w.copy())
            if len(sq) > slow_queries_count:
                sq.pop(0)