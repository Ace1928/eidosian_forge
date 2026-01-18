import sys
import types
import stackless
class FirstSwitch:

    def __init__(self, gr):
        self.gr = gr

    def __call__(self, *args, **kw):
        gr = self.gr
        del gr.switch
        run, gr.run = (gr.run, None)
        t = stackless.tasklet(run)
        gr.t = t
        tasklet_to_greenlet[t] = gr
        t.setup(*args, **kw)
        t.run()