import time
import breezy
import breezy.commands
import breezy.help
import breezy.help_topics
import breezy.osutils
def _dump_text(output_dir, topic, text):
    """Dump text for a topic to a file."""
    topic_id = '{}-{}'.format(topic, 'help')
    filename = breezy.osutils.pathjoin(output_dir, topic_id + '.txt')
    with open(filename, 'wb') as f:
        f.write(text.encode('utf-8'))
    return topic_id