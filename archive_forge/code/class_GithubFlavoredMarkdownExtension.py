from markdown.extensions.nl2br import Nl2BrExtension
from mdx_partial_gfm import PartialGithubFlavoredMarkdownExtension
class GithubFlavoredMarkdownExtension(PartialGithubFlavoredMarkdownExtension):
    """
    An extension that is as compatible as possible with GitHub-flavored
    Markdown (GFM).

    This extension aims to be compatible with the standard GFM that GitHub uses
    for comments and issues. It has all the extensions described in the `GFM
    documentation`_, except for intra-GitHub links to commits, repositories,
    and issues.

    Note that Markdown-formatted gists and files (including READMEs) on GitHub
    use a slightly different variant of GFM. For that, use
    :class:`mdx_partial_gfm.PartialGithubFlavoredMarkdownExtension`.

    .. _GFM documentation: https://guides.github.com/features/mastering-markdown/
    """

    def extendMarkdown(self, md):
        PartialGithubFlavoredMarkdownExtension.extendMarkdown(self, md)
        Nl2BrExtension().extendMarkdown(md)