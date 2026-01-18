from __future__ import print_function, absolute_import, division
import sys
import gc
import time
import weakref
import threading
import greenlet
from . import TestCase
from .leakcheck import fails_leakcheck
from .leakcheck import ignores_leakcheck
from .leakcheck import RUNNING_ON_MANYLINUX
def _check_issue251(self, manually_collect_background=True, explicit_reference_to_switch=False):
    assert gc.is_tracked([])
    HasFinalizerTracksInstances.reset()
    greenlet.getcurrent()
    greenlets_before = self.count_objects(greenlet.greenlet, exact_kind=False)
    background_glet_running = threading.Event()
    background_glet_killed = threading.Event()
    background_greenlets = []

    def background_greenlet():
        jd = HasFinalizerTracksInstances('DELETING STACK OBJECT')
        greenlet._greenlet.set_thread_local('test_leaks_key', HasFinalizerTracksInstances('DELETING THREAD STATE'))
        if explicit_reference_to_switch:
            s = greenlet.getcurrent().parent.switch
            s([jd])
        else:
            greenlet.getcurrent().parent.switch([jd])
    bg_main_wrefs = []

    def background_thread():
        glet = greenlet.greenlet(background_greenlet)
        bg_main_wrefs.append(weakref.ref(glet.parent))
        background_greenlets.append(glet)
        glet.switch()
        del glet
        background_glet_running.set()
        background_glet_killed.wait(10)
        if manually_collect_background:
            greenlet.getcurrent()
    t = threading.Thread(target=background_thread)
    t.start()
    background_glet_running.wait(10)
    greenlet.getcurrent()
    lists_before = self.count_objects(list, exact_kind=True)
    assert len(background_greenlets) == 1
    self.assertFalse(background_greenlets[0].dead)
    del background_greenlets[:]
    background_glet_killed.set()
    t.join(10)
    del t
    self.wait_for_pending_cleanups()
    lists_after = self.count_objects(list, exact_kind=True)
    greenlets_after = self.count_objects(greenlet.greenlet, exact_kind=False)
    self.assertLessEqual(lists_after, lists_before)
    if not explicit_reference_to_switch and greenlet._greenlet.get_clocks_used_doing_optional_cleanup() is not None:
        self.assertEqual(greenlets_after, greenlets_before)
        if manually_collect_background:
            self.assertEqual(HasFinalizerTracksInstances.EXTANT_INSTANCES, set())
    else:
        pass
    if greenlet._greenlet.get_clocks_used_doing_optional_cleanup() is not None:
        self.assertClocksUsed()