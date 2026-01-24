#!/bin/bash
# Agent Forge bash completions

_agent_forge_completions() {
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    case "$prev" in
        eidctl|agent)
            COMPREPLY=($(compgen -W "state journal goals steps runs --help" -- "$cur"))
            ;;
        state)
            COMPREPLY=($(compgen -W "snapshot migrate save diff --help" -- "$cur"))
            ;;
        journal)
            COMPREPLY=($(compgen -W "add list rotate --help" -- "$cur"))
            ;;
        goals)
            COMPREPLY=($(compgen -W "add list --help" -- "$cur"))
            ;;
        steps)
            COMPREPLY=($(compgen -W "list --goal-id --help" -- "$cur"))
            ;;
        runs)
            COMPREPLY=($(compgen -W "list show --help" -- "$cur"))
            ;;
    esac
}

complete -F _agent_forge_completions eidctl
complete -F _agent_forge_completions agent
