import gyp.easy_xml as easy_xml
def AddToolFile(self, path):
    """Adds a tool file to the project.

    Args:
      path: Relative path from project to tool file.
    """
    self.tool_files_section.append(['ToolFile', {'RelativePath': path}])