from parlai.core.agents import Agent
from parlai.core.agents import create_agent
import regex
def _split_doc(self, doc):
    """
        Given a doc, split it into chunks (by paragraph).
        """
    GROUP_LENGTH = 0
    docs = []
    curr = []
    curr_len = 0
    for split in regex.split('\\n+', doc):
        split = split.strip()
        if len(split) == 0:
            continue
        if len(curr) > 0 and curr_len + len(split) > GROUP_LENGTH:
            docs.append(' '.join(curr))
            curr = []
            curr_len = 0
        curr.append(split)
        curr_len += len(split)
    if len(curr) > 0:
        docs.append(' '.join(curr))
    return docs