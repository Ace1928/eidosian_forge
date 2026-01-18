from __future__ import print_function
import contextlib
import cProfile
import gc
import inspect
import os
import re
import sys
import threading
import time
import traceback
import types
import warnings
import weakref
from time import perf_counter
from numpy import ndarray
from .Qt import QT_LIB, QtCore
from .util import cprint
from .util.mutex import Mutex
class ObjTracker(object):
    """
    Tracks all objects under the sun, reporting the changes between snapshots: what objects are created, deleted, and persistent.
    This class is very useful for tracking memory leaks. The class goes to great (but not heroic) lengths to avoid tracking 
    its own internal objects.
    
    Example:
        ot = ObjTracker()   # takes snapshot of currently existing objects
           ... do stuff ...
        ot.diff()           # prints lists of objects created and deleted since ot was initialized
           ... do stuff ...
        ot.diff()           # prints lists of objects created and deleted since last call to ot.diff()
                            # also prints list of items that were created since initialization AND have not been deleted yet
                            #   (if done correctly, this list can tell you about objects that were leaked)
           
        arrays = ot.findPersistent('ndarray')  ## returns all objects matching 'ndarray' (string match, not instance checking)
                                               ## that were considered persistent when the last diff() was run
                                               
        describeObj(arrays[0])    ## See if we can determine who has references to this array
    """
    allObjs = {}
    allObjs[id(allObjs)] = None

    def __init__(self):
        self.startRefs = {}
        self.startCount = {}
        self.newRefs = {}
        self.persistentRefs = {}
        self.objTypes = {}
        ObjTracker.allObjs[id(self)] = None
        self.objs = [self.__dict__, self.startRefs, self.startCount, self.newRefs, self.persistentRefs, self.objTypes]
        self.objs.append(self.objs)
        for v in self.objs:
            ObjTracker.allObjs[id(v)] = None
        self.start()

    def findNew(self, regex):
        """Return all objects matching regex that were considered 'new' when the last diff() was run."""
        return self.findTypes(self.newRefs, regex)

    def findPersistent(self, regex):
        """Return all objects matching regex that were considered 'persistent' when the last diff() was run."""
        return self.findTypes(self.persistentRefs, regex)

    def start(self):
        """
        Remember the current set of objects as the comparison for all future calls to diff()
        Called automatically on init, but can be called manually as well.
        """
        refs, count, objs = self.collect()
        for r in self.startRefs:
            self.forgetRef(self.startRefs[r])
        self.startRefs.clear()
        self.startRefs.update(refs)
        for r in refs:
            self.rememberRef(r)
        self.startCount.clear()
        self.startCount.update(count)

    def diff(self, **kargs):
        """
        Compute all differences between the current object set and the reference set.
        Print a set of reports for created, deleted, and persistent objects
        """
        refs, count, objs = self.collect()
        delRefs = {}
        for i in list(self.startRefs.keys()):
            if i not in refs:
                delRefs[i] = self.startRefs[i]
                del self.startRefs[i]
                self.forgetRef(delRefs[i])
        for i in list(self.newRefs.keys()):
            if i not in refs:
                delRefs[i] = self.newRefs[i]
                del self.newRefs[i]
                self.forgetRef(delRefs[i])
        persistentRefs = {}
        createRefs = {}
        for o in refs:
            if o not in self.startRefs:
                if o not in self.newRefs:
                    createRefs[o] = refs[o]
                else:
                    persistentRefs[o] = refs[o]
        for r in self.newRefs:
            self.forgetRef(self.newRefs[r])
        self.newRefs.clear()
        self.newRefs.update(persistentRefs)
        self.newRefs.update(createRefs)
        for r in self.newRefs:
            self.rememberRef(self.newRefs[r])
        self.persistentRefs.clear()
        self.persistentRefs.update(persistentRefs)
        print('----------- Count changes since start: ----------')
        c1 = count.copy()
        for k in self.startCount:
            c1[k] = c1.get(k, 0) - self.startCount[k]
        typs = list(c1.keys())
        typs.sort(key=lambda a: c1[a])
        for t in typs:
            if c1[t] == 0:
                continue
            num = '%d' % c1[t]
            print('  ' + num + ' ' * (10 - len(num)) + str(t))
        print('-----------  %d Deleted since last diff: ------------' % len(delRefs))
        self.report(delRefs, objs, **kargs)
        print('-----------  %d Created since last diff: ------------' % len(createRefs))
        self.report(createRefs, objs, **kargs)
        print('-----------  %d Created since start (persistent): ------------' % len(persistentRefs))
        self.report(persistentRefs, objs, **kargs)

    def __del__(self):
        self.startRefs.clear()
        self.startCount.clear()
        self.newRefs.clear()
        self.persistentRefs.clear()
        del ObjTracker.allObjs[id(self)]
        for v in self.objs:
            del ObjTracker.allObjs[id(v)]

    @classmethod
    def isObjVar(cls, o):
        return type(o) is cls or id(o) in cls.allObjs

    def collect(self):
        print('Collecting list of all objects...')
        gc.collect()
        objs = get_all_objects()
        frame = sys._getframe()
        del objs[id(frame)]
        del objs[id(frame.f_code)]
        ignoreTypes = [int]
        refs = {}
        count = {}
        for k in objs:
            o = objs[k]
            typ = type(o)
            oid = id(o)
            if ObjTracker.isObjVar(o) or typ in ignoreTypes:
                continue
            try:
                ref = weakref.ref(o)
            except:
                ref = None
            refs[oid] = ref
            typ = type(o)
            typStr = typeStr(o)
            self.objTypes[oid] = typStr
            ObjTracker.allObjs[id(typStr)] = None
            count[typ] = count.get(typ, 0) + 1
        print('All objects: %d   Tracked objects: %d' % (len(objs), len(refs)))
        return (refs, count, objs)

    def forgetRef(self, ref):
        if ref is not None:
            del ObjTracker.allObjs[id(ref)]

    def rememberRef(self, ref):
        if ref is not None:
            ObjTracker.allObjs[id(ref)] = None

    def lookup(self, oid, ref, objs=None):
        if ref is None or ref() is None:
            try:
                obj = lookup(oid, objects=objs)
            except:
                obj = None
        else:
            obj = ref()
        return obj

    def report(self, refs, allobjs=None, showIDs=False):
        if allobjs is None:
            allobjs = get_all_objects()
        count = {}
        rev = {}
        for oid in refs:
            obj = self.lookup(oid, refs[oid], allobjs)
            if obj is None:
                typ = '[del] ' + self.objTypes[oid]
            else:
                typ = typeStr(obj)
            if typ not in rev:
                rev[typ] = []
            rev[typ].append(oid)
            c = count.get(typ, [0, 0])
            count[typ] = [c[0] + 1, c[1] + objectSize(obj)]
        typs = list(count.keys())
        typs.sort(key=lambda a: count[a][1])
        for t in typs:
            line = '  %d\t%d\t%s' % (count[t][0], count[t][1], t)
            if showIDs:
                line += '\t' + ','.join(map(str, rev[t]))
            print(line)

    def findTypes(self, refs, regex):
        allObjs = get_all_objects()
        objs = []
        r = re.compile(regex)
        for k in refs:
            if r.search(self.objTypes[k]):
                objs.append(self.lookup(k, refs[k], allObjs))
        return objs