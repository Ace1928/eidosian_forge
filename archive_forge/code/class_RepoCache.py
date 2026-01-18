import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
class RepoCache:
    """ The Repocache manages discovered repository paths
    and their revisions.  If inside a timeout the cache
    will even return the revision of the root.
    """
    timeout = 20

    def __init__(self):
        self.repos = []

    def clear(self):
        self.repos = []

    def put(self, url, rev, timestamp=None):
        if rev is None:
            return
        if timestamp is None:
            timestamp = time.time()
        for entry in self.repos:
            if url == entry.url:
                entry.timestamp = timestamp
                entry.rev = rev
                break
        else:
            entry = RepoEntry(url, rev, timestamp)
            self.repos.append(entry)

    def get(self, url):
        now = time.time()
        for entry in self.repos:
            if url.startswith(entry.url):
                if now < entry.timestamp + self.timeout:
                    return (entry.url, entry.rev)
                return (entry.url, -1)
        return (url, -1)