import ast
import asyncio
import code
import concurrent.futures
import inspect
import sys
import threading
import types
import warnings
from . import futures
class AsyncIOInteractiveConsole(code.InteractiveConsole):

    def __init__(self, locals, loop):
        super().__init__(locals)
        self.compile.compiler.flags |= ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
        self.loop = loop

    def runcode(self, code):
        future = concurrent.futures.Future()

        def callback():
            global repl_future
            global repl_future_interrupted
            repl_future = None
            repl_future_interrupted = False
            func = types.FunctionType(code, self.locals)
            try:
                coro = func()
            except SystemExit:
                raise
            except KeyboardInterrupt as ex:
                repl_future_interrupted = True
                future.set_exception(ex)
                return
            except BaseException as ex:
                future.set_exception(ex)
                return
            if not inspect.iscoroutine(coro):
                future.set_result(coro)
                return
            try:
                repl_future = self.loop.create_task(coro)
                futures._chain_future(repl_future, future)
            except BaseException as exc:
                future.set_exception(exc)
        loop.call_soon_threadsafe(callback)
        try:
            return future.result()
        except SystemExit:
            raise
        except BaseException:
            if repl_future_interrupted:
                self.write('\nKeyboardInterrupt\n')
            else:
                self.showtraceback()