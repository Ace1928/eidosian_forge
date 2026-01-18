import time
from .exceptions import EOF, TIMEOUT
class Expecter(object):

    def __init__(self, spawn, searcher, searchwindowsize=-1):
        self.spawn = spawn
        self.searcher = searcher
        if searchwindowsize == -1:
            searchwindowsize = spawn.searchwindowsize
        self.searchwindowsize = searchwindowsize
        self.lookback = None
        if hasattr(searcher, 'longest_string'):
            self.lookback = searcher.longest_string

    def do_search(self, window, freshlen):
        spawn = self.spawn
        searcher = self.searcher
        if freshlen > len(window):
            freshlen = len(window)
        index = searcher.search(window, freshlen, self.searchwindowsize)
        if index >= 0:
            spawn._buffer = spawn.buffer_type()
            spawn._buffer.write(window[searcher.end:])
            spawn.before = spawn._before.getvalue()[0:-(len(window) - searcher.start)]
            spawn._before = spawn.buffer_type()
            spawn._before.write(window[searcher.end:])
            spawn.after = window[searcher.start:searcher.end]
            spawn.match = searcher.match
            spawn.match_index = index
            return index
        elif self.searchwindowsize or self.lookback:
            maintain = self.searchwindowsize or self.lookback
            if spawn._buffer.tell() > maintain:
                spawn._buffer = spawn.buffer_type()
                spawn._buffer.write(window[-maintain:])

    def existing_data(self):
        spawn = self.spawn
        before_len = spawn._before.tell()
        buf_len = spawn._buffer.tell()
        freshlen = before_len
        if before_len > buf_len:
            if not self.searchwindowsize:
                spawn._buffer = spawn.buffer_type()
                window = spawn._before.getvalue()
                spawn._buffer.write(window)
            elif buf_len < self.searchwindowsize:
                spawn._buffer = spawn.buffer_type()
                spawn._before.seek(max(0, before_len - self.searchwindowsize))
                window = spawn._before.read()
                spawn._buffer.write(window)
            else:
                spawn._buffer.seek(max(0, buf_len - self.searchwindowsize))
                window = spawn._buffer.read()
        elif self.searchwindowsize:
            spawn._buffer.seek(max(0, buf_len - self.searchwindowsize))
            window = spawn._buffer.read()
        else:
            window = spawn._buffer.getvalue()
        return self.do_search(window, freshlen)

    def new_data(self, data):
        spawn = self.spawn
        freshlen = len(data)
        spawn._before.write(data)
        if not self.searchwindowsize:
            if self.lookback:
                old_len = spawn._buffer.tell()
                spawn._buffer.write(data)
                spawn._buffer.seek(max(0, old_len - self.lookback))
                window = spawn._buffer.read()
            else:
                spawn._buffer.write(data)
                window = spawn.buffer
        elif len(data) >= self.searchwindowsize or not spawn._buffer.tell():
            window = data[-self.searchwindowsize:]
            spawn._buffer = spawn.buffer_type()
            spawn._buffer.write(window[-self.searchwindowsize:])
        else:
            spawn._buffer.write(data)
            new_len = spawn._buffer.tell()
            spawn._buffer.seek(max(0, new_len - self.searchwindowsize))
            window = spawn._buffer.read()
        return self.do_search(window, freshlen)

    def eof(self, err=None):
        spawn = self.spawn
        spawn.before = spawn._before.getvalue()
        spawn._buffer = spawn.buffer_type()
        spawn._before = spawn.buffer_type()
        spawn.after = EOF
        index = self.searcher.eof_index
        if index >= 0:
            spawn.match = EOF
            spawn.match_index = index
            return index
        else:
            spawn.match = None
            spawn.match_index = None
            msg = str(spawn)
            msg += '\nsearcher: %s' % self.searcher
            if err is not None:
                msg = str(err) + '\n' + msg
            exc = EOF(msg)
            exc.__cause__ = None
            raise exc

    def timeout(self, err=None):
        spawn = self.spawn
        spawn.before = spawn._before.getvalue()
        spawn.after = TIMEOUT
        index = self.searcher.timeout_index
        if index >= 0:
            spawn.match = TIMEOUT
            spawn.match_index = index
            return index
        else:
            spawn.match = None
            spawn.match_index = None
            msg = str(spawn)
            msg += '\nsearcher: %s' % self.searcher
            if err is not None:
                msg = str(err) + '\n' + msg
            exc = TIMEOUT(msg)
            exc.__cause__ = None
            raise exc

    def errored(self):
        spawn = self.spawn
        spawn.before = spawn._before.getvalue()
        spawn.after = None
        spawn.match = None
        spawn.match_index = None

    def expect_loop(self, timeout=-1):
        """Blocking expect"""
        spawn = self.spawn
        if timeout is not None:
            end_time = time.time() + timeout
        try:
            idx = self.existing_data()
            if idx is not None:
                return idx
            while True:
                if timeout is not None and timeout < 0:
                    return self.timeout()
                incoming = spawn.read_nonblocking(spawn.maxread, timeout)
                if self.spawn.delayafterread is not None:
                    time.sleep(self.spawn.delayafterread)
                idx = self.new_data(incoming)
                if idx is not None:
                    return idx
                if timeout is not None:
                    timeout = end_time - time.time()
        except EOF as e:
            return self.eof(e)
        except TIMEOUT as e:
            return self.timeout(e)
        except:
            self.errored()
            raise