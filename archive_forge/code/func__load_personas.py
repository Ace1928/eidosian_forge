import json
import random
from parlai.tasks.blended_skill_talk.agents import raw_data_path, safe_personas_path
from parlai.tasks.interactive.worlds import InteractiveWorld as InteractiveBaseWorld
from parlai.tasks.self_chat.worlds import SelfChatWorld as SelfChatBaseWorld
def _load_personas(opt):
    print('[ loading personas.. ]')
    if opt.get('include_personas', True):
        print('\n  [NOTE: In the BST paper both partners have a persona.\n' + '         You can choose to ignore yours, the model never sees it.\n' + '         In the Blender paper, this was not used for humans.\n' + '         You can also turn personas off with --include-personas False]\n')
    fname = raw_data_path(opt)
    with open(fname) as json_file:
        data = json.load(json_file)
    if opt.get('include_personas', True) and opt.get('safe_personas_only', True):
        save_personas_path = safe_personas_path(opt)
        with open(save_personas_path, 'r') as f:
            raw_safe_persona_groups = [line.strip() for line in f.readlines()]
        safe_persona_strings = set()
        for group in raw_safe_persona_groups:
            safe_group = [_standardize(string) for string in group.split('|')]
            safe_persona_strings.update(set(safe_group))
    contexts = []
    for d in data:
        context1 = []
        context2 = []
        if opt.get('include_personas', True):
            if opt.get('safe_personas_only', True):
                personas_are_safe = all((_standardize(persona_string) in safe_persona_strings for persona in d['personas'] for persona_string in persona))
                if not personas_are_safe:
                    continue
            context1.append('your persona: ' + d['personas'][0][0])
            context1.append('your persona: ' + d['personas'][0][1])
            context2.append('your persona: ' + d['personas'][1][0])
            context2.append('your persona: ' + d['personas'][1][1])
        if d['context_dataset'] == 'wizard_of_wikipedia':
            context1.append(d['additional_context'])
            context2.append(d['additional_context'])
        if opt.get('include_initial_utterances', True):
            context1.append(d['free_turker_utterance'])
            context2.append(d['free_turker_utterance'])
            context1.append(d['guided_turker_utterance'])
            context2.append(d['guided_turker_utterance'])
        c1 = '\n'.join(context1)
        c2 = '\n'.join(context2)
        contexts.append([c1, c2])
    return contexts