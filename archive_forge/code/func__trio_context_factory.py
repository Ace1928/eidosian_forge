import dns._features
import dns.asyncbackend
def _trio_context_factory():
    return trio.open_nursery()