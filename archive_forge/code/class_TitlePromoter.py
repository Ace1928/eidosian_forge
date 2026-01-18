import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
class TitlePromoter(Transform):
    """
    Abstract base class for DocTitle and SectionSubTitle transforms.
    """

    def promote_title(self, node):
        """
        Transform the following tree::

            <node>
                <section>
                    <title>
                    ...

        into ::

            <node>
                <title>
                ...

        `node` is normally a document.
        """
        if not isinstance(node, nodes.Element):
            raise TypeError('node must be of Element-derived type.')
        assert not (len(node) and isinstance(node[0], nodes.title))
        section, index = self.candidate_index(node)
        if index is None:
            return None
        node.update_all_atts_concatenating(section, True, True)
        node[:] = section[:1] + node[:index] + section[1:]
        assert isinstance(node[0], nodes.title)
        return 1

    def promote_subtitle(self, node):
        """
        Transform the following node tree::

            <node>
                <title>
                <section>
                    <title>
                    ...

        into ::

            <node>
                <title>
                <subtitle>
                ...
        """
        if not isinstance(node, nodes.Element):
            raise TypeError('node must be of Element-derived type.')
        subsection, index = self.candidate_index(node)
        if index is None:
            return None
        subtitle = nodes.subtitle()
        subtitle.update_all_atts_concatenating(subsection, True, True)
        subtitle[:] = subsection[0][:]
        node[:] = node[:1] + [subtitle] + node[1:index] + subsection[1:]
        return 1

    def candidate_index(self, node):
        """
        Find and return the promotion candidate and its index.

        Return (None, None) if no valid candidate was found.
        """
        index = node.first_child_not_matching_class(nodes.PreBibliographic)
        if index is None or len(node) > index + 1 or (not isinstance(node[index], nodes.section)):
            return (None, None)
        else:
            return (node[index], index)