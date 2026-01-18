import copy
import os
import sys
from io import BytesIO
from xml.dom.minidom import getDOMImplementation
from twisted.internet import address, reactor
from twisted.logger import Logger
from twisted.persisted import styles
from twisted.spread import pb
from twisted.spread.banana import SIZE_LIMIT
from twisted.web import http, resource, server, static, util
from twisted.web.http_headers import Headers
class UserDirectory(resource.Resource):
    """
    A resource which lists available user resources and serves them as
    children.

    @ivar _pwd: An object like L{pwd} which is used to enumerate users and
        their home directories.
    """
    userDirName = 'public_html'
    userSocketName = '.twistd-web-pb'
    template = '\n<html>\n    <head>\n    <title>twisted.web.distrib.UserDirectory</title>\n    <style>\n\n    a\n    {\n        font-family: Lucida, Verdana, Helvetica, Arial, sans-serif;\n        color: #369;\n        text-decoration: none;\n    }\n\n    th\n    {\n        font-family: Lucida, Verdana, Helvetica, Arial, sans-serif;\n        font-weight: bold;\n        text-decoration: none;\n        text-align: left;\n    }\n\n    pre, code\n    {\n        font-family: "Courier New", Courier, monospace;\n    }\n\n    p, body, td, ol, ul, menu, blockquote, div\n    {\n        font-family: Lucida, Verdana, Helvetica, Arial, sans-serif;\n        color: #000;\n    }\n    </style>\n    </head>\n\n    <body>\n    <h1>twisted.web.distrib.UserDirectory</h1>\n\n    %(users)s\n</body>\n</html>\n'

    def __init__(self, userDatabase=None):
        resource.Resource.__init__(self)
        if userDatabase is None:
            userDatabase = pwd
        self._pwd = userDatabase

    def _users(self):
        """
        Return a list of two-tuples giving links to user resources and text to
        associate with those links.
        """
        users = []
        for user in self._pwd.getpwall():
            name, passwd, uid, gid, gecos, dir, shell = user
            realname = gecos.split(',')[0]
            if not realname:
                realname = name
            if os.path.exists(os.path.join(dir, self.userDirName)):
                users.append((name, realname + ' (file)'))
            twistdsock = os.path.join(dir, self.userSocketName)
            if os.path.exists(twistdsock):
                linkName = name + '.twistd'
                users.append((linkName, realname + ' (twistd)'))
        return users

    def render_GET(self, request):
        """
        Render as HTML a listing of all known users with links to their
        personal resources.
        """
        domImpl = getDOMImplementation()
        newDoc = domImpl.createDocument(None, 'ul', None)
        listing = newDoc.documentElement
        for link, text in self._users():
            linkElement = newDoc.createElement('a')
            linkElement.setAttribute('href', link + '/')
            textNode = newDoc.createTextNode(text)
            linkElement.appendChild(textNode)
            item = newDoc.createElement('li')
            item.appendChild(linkElement)
            listing.appendChild(item)
        htmlDoc = self.template % {'users': listing.toxml()}
        return htmlDoc.encode('utf-8')

    def getChild(self, name, request):
        if name == b'':
            return self
        td = b'.twistd'
        if name.endswith(td):
            username = name[:-len(td)]
            sub = 1
        else:
            username = name
            sub = 0
        try:
            pw_name, pw_passwd, pw_uid, pw_gid, pw_gecos, pw_dir, pw_shell = self._pwd.getpwnam(username.decode(sys.getfilesystemencoding()))
        except KeyError:
            return resource._UnsafeNoResource()
        if sub:
            twistdsock = os.path.join(pw_dir, self.userSocketName)
            rs = ResourceSubscription('unix', twistdsock)
            self.putChild(name, rs)
            return rs
        else:
            path = os.path.join(pw_dir, self.userDirName)
            if not os.path.exists(path):
                return resource._UnsafeNoResource()
            return static.File(path)