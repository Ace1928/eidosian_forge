import glob
import operator
import os
import shutil
import sys
import tempfile
from incremental import Version
from twisted.python import release
from twisted.python._release import (
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
class ProjectTests(ExternalTempdirTestCase):
    """
    There is a first-class representation of a project.
    """

    def assertProjectsEqual(self, observedProjects, expectedProjects):
        """
        Assert that two lists of L{Project}s are equal.
        """
        self.assertEqual(len(observedProjects), len(expectedProjects))
        observedProjects = sorted(observedProjects, key=operator.attrgetter('directory'))
        expectedProjects = sorted(expectedProjects, key=operator.attrgetter('directory'))
        for observed, expected in zip(observedProjects, expectedProjects):
            self.assertEqual(observed.directory, expected.directory)

    def makeProject(self, version, baseDirectory=None):
        """
        Make a Twisted-style project in the given base directory.

        @param baseDirectory: The directory to create files in
            (as a L{FilePath).
        @param version: The version information for the project.
        @return: L{Project} pointing to the created project.
        """
        if baseDirectory is None:
            baseDirectory = FilePath(self.mktemp())
        segments = version[0].split('.')
        directory = baseDirectory
        for segment in segments:
            directory = directory.child(segment)
            if not directory.exists():
                directory.createDirectory()
            directory.child('__init__.py').setContent(b'')
        directory.child('newsfragments').createDirectory()
        directory.child('_version.py').setContent(genVersion(*version).encode())
        return Project(directory)

    def makeProjects(self, *versions):
        """
        Create a series of projects underneath a temporary base directory.

        @return: A L{FilePath} for the base directory.
        """
        baseDirectory = FilePath(self.mktemp())
        for version in versions:
            self.makeProject(version, baseDirectory)
        return baseDirectory

    def test_getVersion(self):
        """
        Project objects know their version.
        """
        version = ('twisted', 2, 1, 0)
        project = self.makeProject(version)
        self.assertEqual(project.getVersion(), Version(*version))

    def test_repr(self):
        """
        The representation of a Project is Project(directory).
        """
        foo = Project(FilePath('bar'))
        self.assertEqual(repr(foo), 'Project(%r)' % foo.directory)

    def test_findTwistedStyleProjects(self):
        """
        findTwistedStyleProjects finds all projects underneath a particular
        directory. A 'project' is defined by the existence of a 'newsfragments'
        directory and is returned as a Project object.
        """
        baseDirectory = self.makeProjects(('foo', 2, 3, 0), ('foo.bar', 0, 7, 4))
        projects = findTwistedProjects(baseDirectory)
        self.assertProjectsEqual(projects, [Project(baseDirectory.child('foo')), Project(baseDirectory.child('foo').child('bar'))])