import greenlet

Uses a trace function to switch greenlets at unexpected times.

In the trace function, we switch from the current greenlet to another
greenlet, which switches
