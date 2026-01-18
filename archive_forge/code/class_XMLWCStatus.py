import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
class XMLWCStatus(WCStatus):

    def fromstring(data, rootwcpath, rev=None, modrev=None, author=None):
        """ parse 'data' (XML string as outputted by svn st) into a status obj
        """
        rootstatus = WCStatus(rootwcpath, rev, modrev, author)
        update_rev = None
        minidom, ExpatError = importxml()
        try:
            doc = minidom.parseString(data)
        except ExpatError:
            e = sys.exc_info()[1]
            raise ValueError(str(e))
        urevels = doc.getElementsByTagName('against')
        if urevels:
            rootstatus.update_rev = urevels[-1].getAttribute('revision')
        for entryel in doc.getElementsByTagName('entry'):
            path = entryel.getAttribute('path')
            statusel = entryel.getElementsByTagName('wc-status')[0]
            itemstatus = statusel.getAttribute('item')
            if itemstatus == 'unversioned':
                wcpath = rootwcpath.join(path, abs=1)
                rootstatus.unknown.append(wcpath)
                continue
            elif itemstatus == 'external':
                wcpath = rootwcpath.__class__(rootwcpath.localpath.join(path, abs=1), auth=rootwcpath.auth)
                rootstatus.external.append(wcpath)
                continue
            elif itemstatus == 'ignored':
                wcpath = rootwcpath.join(path, abs=1)
                rootstatus.ignored.append(wcpath)
                continue
            elif itemstatus == 'incomplete':
                wcpath = rootwcpath.join(path, abs=1)
                rootstatus.incomplete.append(wcpath)
                continue
            rev = statusel.getAttribute('revision')
            if itemstatus == 'added' or itemstatus == 'none':
                rev = '0'
                modrev = '?'
                author = '?'
                date = ''
            elif itemstatus == 'replaced':
                pass
            else:
                commitel = entryel.getElementsByTagName('commit')[0]
                if commitel:
                    modrev = commitel.getAttribute('revision')
                    author = ''
                    author_els = commitel.getElementsByTagName('author')
                    if author_els:
                        for c in author_els[0].childNodes:
                            author += c.nodeValue
                    date = ''
                    for c in commitel.getElementsByTagName('date')[0].childNodes:
                        date += c.nodeValue
            wcpath = rootwcpath.join(path, abs=1)
            assert itemstatus != 'modified' or wcpath.check(file=1), "did't expect a directory with changed content here"
            itemattrname = {'normal': 'unchanged', 'unversioned': 'unknown', 'conflicted': 'conflict', 'none': 'added'}.get(itemstatus, itemstatus)
            attr = getattr(rootstatus, itemattrname)
            attr.append(wcpath)
            propsstatus = statusel.getAttribute('props')
            if propsstatus not in ('none', 'normal'):
                rootstatus.prop_modified.append(wcpath)
            if wcpath == rootwcpath:
                rootstatus.rev = rev
                rootstatus.modrev = modrev
                rootstatus.author = author
                rootstatus.date = date
            rstatusels = entryel.getElementsByTagName('repos-status')
            if rstatusels:
                rstatusel = rstatusels[0]
                ritemstatus = rstatusel.getAttribute('item')
                if ritemstatus in ('added', 'modified'):
                    rootstatus.update_available.append(wcpath)
            lockels = entryel.getElementsByTagName('lock')
            if len(lockels):
                rootstatus.locked.append(wcpath)
        return rootstatus
    fromstring = staticmethod(fromstring)