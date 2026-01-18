from parlai.core.params import ParlaiParser
from parlai.scripts.eval_model import eval_model
from parlai.zoo.wizard_of_wikipedia.full_dialogue_retrieval_model import download
from projects.wizard_of_wikipedia.wizard_transformer_ranker.wizard_transformer_ranker import (
Evaluate pre-trained retrieval model on the full Wizard Dialogue task.

NOTE: Metrics here differ slightly to those reported in the paper as a result
of code changes.

Results on seen test set:
Hits@1/100: 86.7

Results on unseen test set (run with flag
`-t wizard_of_wikipedia:WizardDialogKnowledge:topic_split`):
Hits@1/100: 68.96
