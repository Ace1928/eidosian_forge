from cherrypy.process import plugins
class TestAutoreloader:

    def test_file_for_file_module_when_None(self):
        """No error when module.__file__ is None.
        """

        class test_module:
            __file__ = None
        assert plugins.Autoreloader._file_for_file_module(test_module) is None