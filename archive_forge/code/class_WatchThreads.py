import sys
import cgi
import time
import traceback
from io import StringIO
from thread import get_ident
from paste import httpexceptions
from paste.request import construct_url, parse_formvars
from paste.util.template import HTMLTemplate, bunch
class WatchThreads(object):
    """
    Application that watches the threads in ``paste.httpserver``,
    showing the length each thread has been working on a request.

    If allow_kill is true, then you can kill errant threads through
    this application.

    This application can expose private information (specifically in
    the environment, like cookies), so it should be protected.
    """

    def __init__(self, allow_kill=False):
        self.allow_kill = allow_kill

    def __call__(self, environ, start_response):
        if 'paste.httpserver.thread_pool' not in environ:
            start_response('403 Forbidden', [('Content-type', 'text/plain')])
            return ['You must use the threaded Paste HTTP server to use this application']
        if environ.get('PATH_INFO') == '/kill':
            return self.kill(environ, start_response)
        else:
            return self.show(environ, start_response)

    def show(self, environ, start_response):
        start_response('200 OK', [('Content-type', 'text/html')])
        form = parse_formvars(environ)
        if form.get('kill'):
            kill_thread_id = form['kill']
        else:
            kill_thread_id = None
        thread_pool = environ['paste.httpserver.thread_pool']
        nworkers = thread_pool.nworkers
        now = time.time()
        workers = thread_pool.worker_tracker.items()
        workers.sort(key=lambda v: v[1][0])
        threads = []
        for thread_id, (time_started, worker_environ) in workers:
            thread = bunch()
            threads.append(thread)
            if worker_environ:
                thread.uri = construct_url(worker_environ)
            else:
                thread.uri = 'unknown'
            thread.thread_id = thread_id
            thread.time_html = format_time(now - time_started)
            thread.uri_short = shorten(thread.uri)
            thread.environ = worker_environ
            thread.traceback = traceback_thread(thread_id)
        page = page_template.substitute(title='Thread Pool Worker Tracker', nworkers=nworkers, actual_workers=len(thread_pool.workers), nworkers_used=len(workers), script_name=environ['SCRIPT_NAME'], kill_thread_id=kill_thread_id, allow_kill=self.allow_kill, threads=threads, this_thread_id=get_ident(), track_threads=thread_pool.track_threads())
        return [page]

    def kill(self, environ, start_response):
        if not self.allow_kill:
            exc = httpexceptions.HTTPForbidden('Killing threads has not been enabled.  Shame on you for trying!')
            return exc(environ, start_response)
        vars = parse_formvars(environ)
        thread_id = int(vars['thread_id'])
        thread_pool = environ['paste.httpserver.thread_pool']
        if thread_id not in thread_pool.worker_tracker:
            exc = httpexceptions.PreconditionFailed('You tried to kill thread %s, but it is not working on any requests' % thread_id)
            return exc(environ, start_response)
        thread_pool.kill_worker(thread_id)
        script_name = environ['SCRIPT_NAME'] or '/'
        exc = httpexceptions.HTTPFound(headers=[('Location', script_name + '?kill=%s' % thread_id)])
        return exc(environ, start_response)