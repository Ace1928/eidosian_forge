from matplotlib import _api, backend_tools, cbook, widgets

        Return the tool object with the given name.

        For convenience, this passes tool objects through.

        Parameters
        ----------
        name : str or `.ToolBase`
            Name of the tool, or the tool itself.
        warn : bool, default: True
            Whether a warning should be emitted it no tool with the given name
            exists.

        Returns
        -------
        `.ToolBase` or None
            The tool or None if no tool with the given name exists.
        