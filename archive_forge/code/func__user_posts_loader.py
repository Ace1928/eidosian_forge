from __future__ import annotations
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _user_posts_loader(self, search_query: str, category: str, reddit: praw.reddit.Reddit) -> Iterable[Document]:
    user = reddit.redditor(search_query)
    method = getattr(user.submissions, category)
    cat_posts = method(limit=self.number_posts)
    'Format reddit posts into a string.'
    for post in cat_posts:
        metadata = {'post_subreddit': post.subreddit_name_prefixed, 'post_category': category, 'post_title': post.title, 'post_score': post.score, 'post_id': post.id, 'post_url': post.url, 'post_author': post.author}
        yield Document(page_content=post.selftext, metadata=metadata)