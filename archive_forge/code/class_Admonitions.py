from docutils import nodes, utils, languages
from docutils.transforms import Transform
class Admonitions(Transform):
    """
    Transform specific admonitions, like this:

        <note>
            <paragraph>
                 Note contents ...

    into generic admonitions, like this::

        <admonition classes="note">
            <title>
                Note
            <paragraph>
                Note contents ...

    The admonition title is localized.
    """
    default_priority = 920

    def apply(self):
        language = languages.get_language(self.document.settings.language_code, self.document.reporter)
        for node in self.document.traverse(nodes.Admonition):
            node_name = node.__class__.__name__
            node['classes'].append(node_name)
            if not isinstance(node, nodes.admonition):
                admonition = nodes.admonition(node.rawsource, *node.children, **node.attributes)
                title = nodes.title('', language.labels[node_name])
                admonition.insert(0, title)
                node.replace_self(admonition)