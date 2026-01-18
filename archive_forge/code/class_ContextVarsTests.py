from __future__ import print_function
import gc
import sys
import unittest
from functools import partial
from unittest import skipUnless
from unittest import skipIf
from greenlet import greenlet
from greenlet import getcurrent
from . import TestCase
@skipUnless(Context is not None, 'ContextVar not supported')
class ContextVarsTests(TestCase):

    def _new_ctx_run(self, *args, **kwargs):
        return copy_context().run(*args, **kwargs)

    def _increment(self, greenlet_id, callback, counts, expect):
        ctx_var = ID_VAR
        if expect is None:
            self.assertIsNone(ctx_var.get())
        else:
            self.assertEqual(ctx_var.get(), expect)
        ctx_var.set(greenlet_id)
        for _ in range(2):
            counts[ctx_var.get()] += 1
            callback()

    def _test_context(self, propagate_by):
        ID_VAR.set(0)
        callback = getcurrent().switch
        counts = dict(((i, 0) for i in range(5)))
        lets = [greenlet(partial(partial(copy_context().run, self._increment) if propagate_by == 'run' else self._increment, greenlet_id=i, callback=callback, counts=counts, expect=i - 1 if propagate_by == 'share' else 0 if propagate_by in ('set', 'run') else None)) for i in range(1, 5)]
        for let in lets:
            if propagate_by == 'set':
                let.gr_context = copy_context()
            elif propagate_by == 'share':
                let.gr_context = getcurrent().gr_context
        for i in range(2):
            counts[ID_VAR.get()] += 1
            for let in lets:
                let.switch()
        if propagate_by == 'run':
            for let in reversed(lets):
                let.switch()
        else:
            for let in lets:
                let.switch()
        for let in lets:
            self.assertTrue(let.dead)
            if propagate_by == 'run':
                self.assertIsNone(let.gr_context)
            else:
                self.assertIsNotNone(let.gr_context)
        if propagate_by == 'share':
            self.assertEqual(counts, {0: 1, 1: 1, 2: 1, 3: 1, 4: 6})
        else:
            self.assertEqual(set(counts.values()), set([2]))

    def test_context_propagated_by_context_run(self):
        self._new_ctx_run(self._test_context, 'run')

    def test_context_propagated_by_setting_attribute(self):
        self._new_ctx_run(self._test_context, 'set')

    def test_context_not_propagated(self):
        self._new_ctx_run(self._test_context, None)

    def test_context_shared(self):
        self._new_ctx_run(self._test_context, 'share')

    def test_break_ctxvars(self):
        let1 = greenlet(copy_context().run)
        let2 = greenlet(copy_context().run)
        let1.switch(getcurrent().switch)
        let2.switch(getcurrent().switch)
        let1.switch()

    def test_not_broken_if_using_attribute_instead_of_context_run(self):
        let1 = greenlet(getcurrent().switch)
        let2 = greenlet(getcurrent().switch)
        let1.gr_context = copy_context()
        let2.gr_context = copy_context()
        let1.switch()
        let2.switch()
        let1.switch()
        let2.switch()

    def test_context_assignment_while_running(self):
        ID_VAR.set(None)

        def target():
            self.assertIsNone(ID_VAR.get())
            self.assertIsNone(gr.gr_context)
            ID_VAR.set(1)
            self.assertIsInstance(gr.gr_context, Context)
            self.assertEqual(ID_VAR.get(), 1)
            self.assertEqual(gr.gr_context[ID_VAR], 1)
            old_context = gr.gr_context
            gr.gr_context = None
            self.assertIsNone(ID_VAR.get())
            self.assertIsNone(gr.gr_context)
            ID_VAR.set(2)
            self.assertIsInstance(gr.gr_context, Context)
            self.assertEqual(ID_VAR.get(), 2)
            self.assertEqual(gr.gr_context[ID_VAR], 2)
            new_context = gr.gr_context
            getcurrent().parent.switch((old_context, new_context))
            self.assertEqual(ID_VAR.get(), 1)
            gr.gr_context = new_context
            self.assertEqual(ID_VAR.get(), 2)
            getcurrent().parent.switch()
            self.assertIsNone(ID_VAR.get())
            self.assertIsNone(gr.gr_context)
            gr.gr_context = old_context
            self.assertEqual(ID_VAR.get(), 1)
            getcurrent().parent.switch()
            self.assertIsNone(ID_VAR.get())
            self.assertIsNone(gr.gr_context)
        gr = greenlet(target)
        with self.assertRaisesRegex(AttributeError, "can't delete context attribute"):
            del gr.gr_context
        self.assertIsNone(gr.gr_context)
        old_context, new_context = gr.switch()
        self.assertIs(new_context, gr.gr_context)
        self.assertEqual(old_context[ID_VAR], 1)
        self.assertEqual(new_context[ID_VAR], 2)
        self.assertEqual(new_context.run(ID_VAR.get), 2)
        gr.gr_context = old_context
        gr.switch()
        self.assertIs(gr.gr_context, new_context)
        gr.gr_context = None
        gr.switch()
        self.assertIs(gr.gr_context, old_context)
        gr.gr_context = None
        gr.switch()
        self.assertIsNone(gr.gr_context)
        gr = None
        gc.collect()
        self.assertEqual(sys.getrefcount(old_context), 2)
        self.assertEqual(sys.getrefcount(new_context), 2)

    def test_context_assignment_different_thread(self):
        import threading
        VAR_VAR.set(None)
        ctx = Context()
        is_running = threading.Event()
        should_suspend = threading.Event()
        did_suspend = threading.Event()
        should_exit = threading.Event()
        holder = []

        def greenlet_in_thread_fn():
            VAR_VAR.set(1)
            is_running.set()
            should_suspend.wait(10)
            VAR_VAR.set(2)
            getcurrent().parent.switch()
            holder.append(VAR_VAR.get())

        def thread_fn():
            gr = greenlet(greenlet_in_thread_fn)
            gr.gr_context = ctx
            holder.append(gr)
            gr.switch()
            did_suspend.set()
            should_exit.wait(10)
            gr.switch()
            del gr
            greenlet()
        thread = threading.Thread(target=thread_fn, daemon=True)
        thread.start()
        is_running.wait(10)
        gr = holder[0]
        with self.assertRaisesRegex(ValueError, 'running in a different'):
            getattr(gr, 'gr_context')
        with self.assertRaisesRegex(ValueError, 'running in a different'):
            gr.gr_context = None
        should_suspend.set()
        did_suspend.wait(10)
        self.assertIs(gr.gr_context, ctx)
        self.assertEqual(gr.gr_context[VAR_VAR], 2)
        gr.gr_context = None
        should_exit.set()
        thread.join(10)
        self.assertEqual(holder, [gr, None])
        self.assertIsNone(gr.gr_context)
        gr.gr_context = ctx
        self.assertIs(gr.gr_context, ctx)
        del holder[:]
        gr = None
        thread = None

    def test_context_assignment_wrong_type(self):
        g = greenlet()
        with self.assertRaisesRegex(TypeError, 'greenlet context must be a contextvars.Context or None'):
            g.gr_context = self