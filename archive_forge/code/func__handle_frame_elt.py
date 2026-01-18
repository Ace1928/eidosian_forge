import itertools
import os
import re
import sys
import textwrap
import types
from collections import OrderedDict, defaultdict
from itertools import zip_longest
from operator import itemgetter
from pprint import pprint
from nltk.corpus.reader import XMLCorpusReader, XMLCorpusView
from nltk.util import LazyConcatenation, LazyIteratorList, LazyMap
def _handle_frame_elt(self, elt, ignorekeys=[]):
    """Load the info for a Frame from a frame xml file"""
    frinfo = self._load_xml_attributes(AttrDict(), elt)
    frinfo['_type'] = 'frame'
    frinfo['definition'] = ''
    frinfo['definitionMarkup'] = ''
    frinfo['FE'] = PrettyDict()
    frinfo['FEcoreSets'] = []
    frinfo['lexUnit'] = PrettyDict()
    frinfo['semTypes'] = []
    for k in ignorekeys:
        if k in frinfo:
            del frinfo[k]
    for sub in elt:
        if sub.tag.endswith('definition') and 'definition' not in ignorekeys:
            frinfo['definitionMarkup'] = sub.text
            frinfo['definition'] = self._strip_tags(sub.text)
        elif sub.tag.endswith('FE') and 'FE' not in ignorekeys:
            feinfo = self._handle_fe_elt(sub)
            frinfo['FE'][feinfo.name] = feinfo
            feinfo['frame'] = frinfo
        elif sub.tag.endswith('FEcoreSet') and 'FEcoreSet' not in ignorekeys:
            coreset = self._handle_fecoreset_elt(sub)
            frinfo['FEcoreSets'].append(PrettyList((frinfo['FE'][fe.name] for fe in coreset)))
        elif sub.tag.endswith('lexUnit') and 'lexUnit' not in ignorekeys:
            luentry = self._handle_framelexunit_elt(sub)
            if luentry['status'] in self._bad_statuses:
                continue
            luentry['frame'] = frinfo
            luentry['URL'] = self._fnweb_url + '/' + self._lu_dir + '/' + 'lu{}.xml'.format(luentry['ID'])
            luentry['subCorpus'] = Future((lambda lu: lambda: self._lu_file(lu).subCorpus)(luentry))
            luentry['exemplars'] = Future((lambda lu: lambda: self._lu_file(lu).exemplars)(luentry))
            frinfo['lexUnit'][luentry.name] = luentry
            if not self._lu_idx:
                self._buildluindex()
            self._lu_idx[luentry.ID] = luentry
        elif sub.tag.endswith('semType') and 'semTypes' not in ignorekeys:
            semtypeinfo = self._load_xml_attributes(AttrDict(), sub)
            frinfo['semTypes'].append(self.semtype(semtypeinfo.ID))
    frinfo['frameRelations'] = self.frame_relations(frame=frinfo)
    for fe in frinfo.FE.values():
        if fe.requiresFE:
            name, ID = (fe.requiresFE.name, fe.requiresFE.ID)
            fe.requiresFE = frinfo.FE[name]
            assert fe.requiresFE.ID == ID
        if fe.excludesFE:
            name, ID = (fe.excludesFE.name, fe.excludesFE.ID)
            fe.excludesFE = frinfo.FE[name]
            assert fe.excludesFE.ID == ID
    return frinfo